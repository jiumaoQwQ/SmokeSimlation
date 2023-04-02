#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

#include "vdbhelper.h"

void writeDense(std::string name, float *data, int x, int y, int z)
{
    openvdb::CoordBBox bbox(0, 0, 0, x - 1, y - 1, z - 1);
    openvdb::tools::Dense<float,openvdb::tools::LayoutXYZ> dense(bbox, data);
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    openvdb::tools::copyFromDense(dense, grid->tree(), float{0});
    grid->insertMeta("name", openvdb::StringMetadata("density"));

    openvdb::io::File(name).write({grid});
}